from embedding_modifier.trainers.long_latent_model_trainer import LongLatentModelTrainer

trainer = LongLatentModelTrainer(tensor_board=False)

trainer.train(epochs=1000)
