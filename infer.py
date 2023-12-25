import hydra
from omegaconf import DictConfig

import mlops_project.infer_rfr as infer


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    infer.main(cfg)


if __name__ == "__main__":
    main()
