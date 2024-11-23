"""Runs web app with t-SNE visualizations of embeddings and latents"""
from database_management.database_visualizers.tsne_visualizer import TsneVisualizer
from database_management.database_managers.tsne_database_manager import TsneDatabaseManager


# needs TSNE database initialized

tsne_man = TsneDatabaseManager()
dv = TsneVisualizer()
dv.visualize()
