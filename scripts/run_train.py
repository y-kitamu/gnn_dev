"""run_train.py
"""

# import argparse
# from pathlib import Path

# import keras

# import gnn


# def run(config_path: Path):
#     """ """
#     if config_path.exists():
#         with open(config_path, "r") as f:
#             config = gnn.TrainParams.model_validate_json(f.read())
#     else:
#         config = gnn.TrainParams()

#     model = gnn.layers.get_model(config.model_params)
#     loss = gnn.losses.get_loss(config.loss_params)
#     optimizer = gnn.optimizers.get_optimizer(config.optimizer_params)
#     train_dataloader, test_dataloader = gnn.dataloader.get_dataloader(
#         config.train_dataloader_params, config.test_dataloader_params
#     )
#     callbacks = keras.callbacks.CallbackList(
#         [
#             gnn.callbacks.MetricsLogger(),
#         ]
#     )
#     trainer = gnn.BaseTrainer(
#         model=model,
#         loss=loss,
#         optimizer=optimizer,
#         train_dataloader=train_dataloader,
#         test_dataloader=test_dataloader,
#         callbacks=callbacks,
#         config=config,
#     )
#     trainer.train()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=Path, required=True)

#     args = parser.parse_args()

#     run(args.config)
