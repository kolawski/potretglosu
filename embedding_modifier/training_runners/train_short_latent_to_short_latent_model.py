from embedding_modifier.trainers.short_latent_to_short_latent_model_trainer import ShortLatentToShortLatentModelTrainer

trainer = ShortLatentToShortLatentModelTrainer(tensor_board=False, model_version="7", distance_loss_weight=0.8)

trainer.train(epochs=100, load_model=False, checkpoint_interval=30)
trainer.test()
