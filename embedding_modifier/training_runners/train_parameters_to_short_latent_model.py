from embedding_modifier.trainers.parameters_to_short_latent_model_trainer import ParametersToShortLatentModelTrainer

trainer = ParametersToShortLatentModelTrainer(tensor_board=False, model_version="19")

trainer.train(epochs=1000, load_model=True)
trainer.test()
