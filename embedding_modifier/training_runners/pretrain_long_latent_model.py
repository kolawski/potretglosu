from embedding_modifier.pretrainers.long_latent_model_pretrainer import LongLatentModelPretrainer

trainer = LongLatentModelPretrainer(tensor_board=False)

trainer.train(epochs=10)
