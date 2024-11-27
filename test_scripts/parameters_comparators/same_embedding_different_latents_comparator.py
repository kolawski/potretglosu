import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from database_management.database_managers.embedding_database_manager import EMBEDDING_KEY, LATENT_KEY
from settings import SAMPLES_DIRECTORY_PATH, TMP_DIRECTORY_PATH
from test_scripts.parameters_comparators.parameters_comparator import ParametersComparator


EXAMPLE_SAMPLE_PATH = SAMPLES_DIRECTORY_PATH + "/common_voice_pl_20603622.wav"
SAVE_PATH = "/app/results/histograms_same_embedding_random_latents"
PHRASE = "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować."
TITLE_PREFIX = "THE SAME EMBEDDING - RANDOM LATENTS"

class SameEmbeddingsDifferentLatentsComparator(ParametersComparator):
    def __init__(self, save_path=SAVE_PATH, clean_tmp=False, phrase=PHRASE, iterations=200, color="blue"):
        """
        Initialize the comparator with the given parameters.
        :param save_path: Path where results will be saved, defaults to SAVE_PATH
        :type save_path: str, optional
        :param clean_tmp: Flag to indicate whether to clean temporary files, defaults to False
        :type clean_tmp: bool, optional
        :param phrase: Phrase to be used in the comparison, defaults to PHRASE
        :type phrase: str, optional
        :param iterations: Number of iterations for the comparison, defaults to 200
        :type iterations: int, optional
        :param color: Color to be used in the comparison, defaults to "blue"
        :type color: str, optional
        """
        
        super().__init__(save_path, clean_tmp)
    
        # Prepare data
        self.phrase = phrase
        self.file_name_copy_prefix = "/copy_different_latent_"
        self.file_name_original = "/diff_lat_original.wav"
        self.iterations = iterations
        self.color = color

        # Retrieve embedding
        self.example_sample_data = self.edb.filter_data(sample_path=EXAMPLE_SAMPLE_PATH)
        self.embedding = self.example_sample_data[EMBEDDING_KEY].iloc[0]

        # Data to be set
        self.original_parameters = None
        self.parameters_to_compare = None
        self.parameters_to_compare_dict = None

    def _retrieve_original_parameters(self):
        """
        Retrieve and process the original parameters.
        This method retrieves the original latent parameters from the example sample data,
        prints the shapes and values of the embedding and original latent, and processes
        the latent-embedding pair by calling the `process_latent_and_embedding` method.
        """
        
        original_latent = self.example_sample_data[LATENT_KEY].iloc[0]
        print(f"Embedding shape: {self.embedding.shape}")
        print(f"Embedding: {self.embedding}")
        print(f"Original Latent shape: {original_latent.shape}")
        print(f"Original Latent: {original_latent}")

        # Original latent-embedding pair
        path = TMP_DIRECTORY_PATH + self.file_name_original
        self.original_parameters = self.process_latent_and_embedding(self.embedding, original_latent, path, self.phrase)

    def _retrieve_parameters_to_compare(self):
        """
        Retrieve parameters to compare by processing random latents.
        This method retrieves all latent values from the database and processes a random selection of them
        to generate parameters for comparison. The parameters are stored in the `parameters_to_compare` attribute.
        """

        all_latents = self.edb.get_all_values_from_column(LATENT_KEY)
        print(f"All latents shape: {all_latents.shape}")
        print(f"All latents type: {type(all_latents)}")
        print(f"All latents: {all_latents}")
        retrieved_parameters = []
        for a in range(self.iterations):
            random_index = random.randint(0, len(all_latents) - 1)
            random_latent = all_latents.iloc[random_index]
            path = TMP_DIRECTORY_PATH + self.file_name_copy_prefix + str(a) + ".wav"
            parameters = self.process_latent_and_embedding(self.embedding, random_latent, path, self.phrase)
            retrieved_parameters.append(parameters)
        self.parameters_to_compare = retrieved_parameters

    def _generate_histograms(self):
        """
        Generate and save histograms for each parameter in the comparison dictionary.
        This method iterates over the `parameters_to_compare_dict` attribute, calculates the mean and standard deviation
        for each parameter's values, and generates a histogram with a KDE (Kernel Density Estimate) plot. The histogram
        includes a vertical line indicating the original latent value for the parameter. The histogram is saved as a PNG
        file with the parameter name as the filename.
        """

        for key, values in self.parameters_to_compare_dict.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            title = f"{TITLE_PREFIX}; {key}, mean={mean_value:.6f}, std={std_value:.6f}"
            plt.figure(figsize=(10, 6))
            sns.histplot(data=values, bins='auto', kde=True, color=self.color)
            # Add value for original latent
            original_latent_value = self.original_parameters[key]
            plt.axvline(x=original_latent_value, color="red", linestyle="--", linewidth=1,
                        label=f"Original latent value = {original_latent_value:.2f}")
            plt.title(title)
            plt.savefig(f"{SAVE_PATH}/{key}.png")
            plt.close()

    def run_comparison(self):
        """
        Run the comparison process between the original parameters and the parameters to compare.
        This method performs the following steps:
        1. Retrieves the original parameters.
        2. Retrieves the parameters to compare.
        3. Converts the list of dictionaries of parameters to compare into a dictionary of lists.
        4. Generates histograms for the comparison.
        """

        self._retrieve_original_parameters()
        self._retrieve_parameters_to_compare()
        self.parameters_to_compare_dict = self.convert_list_of_dicts_to_dict_of_lists(self.parameters_to_compare)
        self._generate_histograms()
