""" Run one experiment and save metrics locally """
import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from .config import (
    EVALUATE_EVERY_STEPS,
    GROK_ACCURACY_THRESHOLD,
    LEARNING_RATE_WARMUP_STEPS,
    MAX_LEARNING_RATE,
    PROTOCOL_VERSION,
    TRANSFORMER_ADAM_BETAS,
    TRANSFORMER_ADAM_EPSILON,
    ExperimentConfig,
    lambda_at_step,
)
from .data import(
    EQUALS_TOKEN,
    TOKEN_TO_ID,
    ArithmeticData,
    BatchStream,
    build_data,
    full_batch,
)
from .intercomponent_ufo import IntercomponentUFO
from .intracomponent_ufo import IntracomponentUFO
from .model import ArithmeticTransformer, TransformerOutput
UFO = IntercomponentUFO | IntracomponentUFO

def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def learning_rate(step: int) -> float:
    if step <= LEARNING_RATE_WARMUP_STEPS:
        return step / LEARNING_RATE_WARMUP_STEPS * MAX_LEARNING_RATE
    return MAX_LEARNING_RATE

def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = value

def build_transformer_optimizer(model: ArithmeticTransformer, weight_decay: float, device: torch.device) -> torch.optim.AdamW:
    options: dict[str, object] = {}
    if device.type == "cuda":
        options["fused"] = True
    return torch.optim.AdamW(model.parameters(), lr=learning_rate(0), betas=TRANSFORMER_ADAM_BETAS, eps=TRANSFORMER_ADAM_EPSILON, weight_decay=weight_decay, **options)

def task_loss_and_accuracy(output: TransformerOutput, target:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    equals_position = int(torch.nonzero(target[0] == TOKEN_TO_ID[EQUALS_TOKEN]).item())
    answer_target = target[:, equals_position + 1 :]
    answer_logits = output.logits[:, equals_position + 1 :]
    predictions = answer_logits.argmax(dim=-1)
    loss = F.cross_entropy(answer_logits.transpose(1, 2), answer_target)
    accuracy = (predictions == answer_target).all(dim=1).float().mean() * 100
    return loss, accuracy

@torch.no_grad()
def evaluate(model: ArithmeticTransformer, data: ArithmeticData, device: torch.device) -> tuple[float, float]:
    model.eval()
    inputs, target = full_batch(data.validation, device)
    loss, accuracy = task_loss_and_accuracy(model(inputs), target)
    model.train()
    return float(loss.item()), float(accuracy.item())

def build_ufo(config: ExperimentConfig, device: torch.device) -> UFO | None:
    if config.treatment == "baseline":
        return None
    if config.family == "intercomponent":
        return IntercomponentUFO(config, device)
    return IntracomponentUFO(config, device)

def _float_name(value: float) -> str:
    return format(value, "g").replace(".", "p")

def experiment_name(config: ExperimentConfig) -> str:
    condition = ("plain" if config.treatment == "baseline" else f"{config.family}-{config.modality}-{config.treatment}")
    name = (f"seed{config.seed:02d}-{condition}" f"-wd{_float_name(config.weight_decay)}")
    if config.treatment != "baseline":
        name += f"-lambda{_float_name(config.lambda_value)}"
    if config.schedule is not None:
        name += (f"-q{_float_name(config.schedule.start_fraction)}" f"to{_float_name(config.schedule.floor_fraction)}" f"-steps{config.schedule.num_ufo_steps}")
    return name

def experiment_metadata(config: ExperimentConfig, device: torch.device) -> dict[str, object]:
    schedule = config.schedule
    return {
        "protocol": PROTOCOL_VERSION,
        "seed": config.seed,
        "treatment": config.treatment,
        "family": config.family,
        "modality": config.modality,
        "weight_decay": config.weight_decay,
        "max_steps": config.max_steps,
        "lambda": config.lambda_value,
        "schedule_start": (None if schedule is None else schedule.start_fraction),
        "schedule_floor": (None if schedule is None else schedule.floor_fraction),
        "schedule_updates": (None if schedule is None else schedule.num_ufo_steps),
        "model_seed": config.model_seed,
        "ufo_seed": config.ufo_seed,
        "batch_seed": config.batch_seed,
        "data_seed": config.data_seed,
        "device": str(device),
        "torch_version": torch.__version__,
    }

def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def train_experiment(config: ExperimentConfig, output_directory: str | Path = "results", device: str = "auto", max_steps: int | None = None) -> Path:
    resolved_device = resolve_device(device)
    maximum_steps = config.max_steps if max_steps is None else max_steps
    data = build_data(seed=config.data_seed)
    set_seed(config.model_seed)
    model = ArithmeticTransformer().to(resolved_device)
    optimizer = build_transformer_optimizer(model, config.weight_decay, resolved_device)
    batches = BatchStream(data.train, resolved_device, config.batch_seed)
    ufo = build_ufo(config, resolved_device)
    rows = []
    metadata = experiment_metadata(config, resolved_device)
    step = 0
    grok_step: int | None = None
    last_evaluation_step = 0
    latest_lambda = 0.0
    latest_ufo_loss = None
    for inputs, target in batches:
        step += 1
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        eligible = (ufo is not None and (not isinstance(ufo, IntracomponentUFO) or ufo.eligible(len(inputs))))
        output = model(inputs, capture=config.modality if eligible else None)
        task_loss, train_accuracy = task_loss_and_accuracy(output, target)
        reconstruction = None
        effective_lambda = 0.0
        if eligible:
            assert ufo is not None
            reconstruction = ufo.transformer_reconstruction(output, inputs)
            effective_lambda = lambda_at_step(config.lambda_value, step)
            total_loss = (task_loss + reconstruction.loss * effective_lambda)
        else:
            total_loss = task_loss
        if (eligible and isinstance(ufo, IntracomponentUFO) and effective_lambda > 0.0):
            assert reconstruction is not None
            task_loss.backward(retain_graph=True)
            contributions = reconstruction.loss_contributions
            for index, contribution in enumerate(contributions):
                (contribution * effective_lambda).backward(retain_graph=index + 1 < len(contributions))
        else:
            total_loss.backward()
        optimizer.step()

        ufo_loss = None
        if eligible:
            assert ufo is not None and reconstruction is not None
            update_rows = (reconstruction.function_rows if isinstance(ufo, IntracomponentUFO) else reconstruction.normalized_rows)
            ufo_loss = ufo.update(update_rows)
            latest_lambda, latest_ufo_loss = effective_lambda, ufo_loss
        set_learning_rate(optimizer, learning_rate(step))

        if step % EVALUATE_EVERY_STEPS == 0:
            validation_loss, validation_accuracy = evaluate(model, data, resolved_device)
            last_evaluation_step = step
            if validation_accuracy >= GROK_ACCURACY_THRESHOLD:
                grok_step = step
            rows.append(
                {
                    **metadata,
                    "step": step,
                    "learning_rate": current_learning_rate,
                    "lambda_effective": latest_lambda,
                    "task_loss": float(task_loss.detach().item()),
                    "train_accuracy": float(train_accuracy.detach().item()),
                    "ufo_loss": (
                        None
                        if latest_ufo_loss is None
                        else float(latest_ufo_loss.item())
                    ),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "grok_step": grok_step,
                    "event": int(grok_step is not None),
                }
            )
            print(
                f"{experiment_name(config)}: step={step}, "
                f"validation_accuracy={validation_accuracy:.2f}"
            )

        if grok_step is not None or step >= maximum_steps:
            break

    if last_evaluation_step != step:
        validation_loss, validation_accuracy = evaluate(model, data, resolved_device)
        if validation_accuracy >= GROK_ACCURACY_THRESHOLD:
            grok_step = step
        rows.append(
            {
                **metadata,
                "step": step,
                "learning_rate": current_learning_rate,
                "lambda_effective": latest_lambda,
                "task_loss": float(task_loss.detach().item()),
                "train_accuracy": float(train_accuracy.detach().item()),
                "ufo_loss": (
                    None if latest_ufo_loss is None else float(latest_ufo_loss.item())
                ),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "grok_step": grok_step,
                "event": int(grok_step is not None),
            }
        )

    for row in rows:
        row["stop_step"] = step
        row ["censored"] = int(grok_step is None)

    path = (Path(output_directory) / f"{experiment_name(config)}.csv")
    write_metrics(path, rows)
    return path
