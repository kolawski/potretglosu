from embedding_modifier.model_trainer import ModelTrainer

trainer = ModelTrainer(tensor_board=False)

trainer.train(epochs=30)
