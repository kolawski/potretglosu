from embedding_modifier.model_pretrainer import ModelPretrainer

trainer = ModelPretrainer(tensor_board=False)

trainer.train(epochs=5000)
