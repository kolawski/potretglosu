from embedding_modifier.trainers.parameters_to_short_latent_model_trainer import ParametersToShortLatentModelTrainer

trainer = ParametersToShortLatentModelTrainer(tensor_board=False, tensor_board_files_suffix="1")

trainer.train(epochs=500)
trainer.test()
