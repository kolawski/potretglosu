from embedding_modifier.trainers.long_latent_model_trainer import LongLatentModelTrainer

trainer = LongLatentModelTrainer(tensor_board=False, model_version="2", save_normalization_dict=False)

trainer.train(epochs=500, load_model=False)
