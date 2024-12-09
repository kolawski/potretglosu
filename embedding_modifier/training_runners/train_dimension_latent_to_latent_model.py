from embedding_modifier.trainers.dimension_latent_to_latent_model_trainer import DimensionLatentToLatentModelTrainer

trainer = DimensionLatentToLatentModelTrainer(tensor_board=False, model_version="4")

trainer.train(epochs=100, load_model=False)
trainer.test()
