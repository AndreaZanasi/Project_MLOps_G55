import typer
import os
import wandb
from typing import List


def link_model(artifact_path: str, aliases: List[str] = ["staging"]) -> None:
    if artifact_path == "":
        typer.echo("No artifact path provided.")
        return

    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))

    artifact = api.artifact(artifact_path)

    entity = "mlops_g55-org"
    registry_name = "wandb-registry-marine_mammal_registry"
    collection_name = "species_classifier"

    target_path = f"{entity}/{registry_name}/{collection_name}"

    typer.echo(f"Linking {artifact_path} to {target_path}")

    artifact.link(target_path=target_path, aliases=aliases)
    artifact.save()

    typer.echo(f"Success! Model linked to {target_path} with aliases {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
