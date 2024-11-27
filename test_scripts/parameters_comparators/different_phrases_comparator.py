import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from database_management.database_managers.embedding_database_manager import EMBEDDING_KEY, LATENT_KEY
from settings import SAMPLES_DIRECTORY_PATH, TMP_DIRECTORY_PATH
from test_scripts.parameters_comparators.parameters_comparator import ParametersComparator


EXAMPLE_SAMPLE_PATH = SAMPLES_DIRECTORY_PATH + "/common_voice_pl_20603622.wav"
SAVE_PATH = "/app/results/histograms_different_phrases"
PHRASES_CSV_PATH = "/app/Resources/other/polish_sentences.csv"
TITLE_PREFIX = "SAME EMBEDDING AND LATENT - DIFF PHRASES"

class DifferentPhrasesComparator(ParametersComparator):
    def __init__(self, save_path=SAVE_PATH, clean_tmp=False, phrases_csv_file=PHRASES_CSV_PATH, iterations=200, color="gray"):
        """
        Initialize the comparator with the given parameters.
        :param save_path: Path where results will be saved, defaults to SAVE_PATH
        :type save_path: str, optional
        :param clean_tmp: Flag to indicate whether to clean temporary files, defaults to False
        :type clean_tmp: bool, optional
        :param phrases_csv_file: Path to the CSV file containing phrases, defaults to PHRASES_CSV_PATH
        :type phrases_csv_file: str, optional
        :param iterations: Number of iterations for the comparison, defaults to 200
        :type iterations: int, optional
        :param color: Color to be used in the comparison, defaults to "gray"
        :type color: str, optional
        """
        
        super().__init__(save_path, clean_tmp)
    
        # Prepare data
        self.phrases_csv_file = phrases_csv_file
        self.file_name_prefix = "/diff_phrase_version_"
        self.iterations = iterations
        self.color = color

        # Retrieve embedding
        self.example_sample_data = self.edb.filter_data(sample_path=EXAMPLE_SAMPLE_PATH)
        self.embedding = self.example_sample_data[EMBEDDING_KEY].iloc[0]
        self.latent = self.example_sample_data[LATENT_KEY].iloc[0]

        # Data to be set
        self.parameters_to_compare = None
        self.parameters_to_compare_dict = None

    def _retrieve_parameters_to_compare(self):
        """
        Retrieve parameters to compare for each iteration.
        This method iterates over a predefined number of iterations, constructs a file path for each iteration,
        processes latent and embedding parameters, and appends the results to a list. The final list of parameters
        is stored in the `parameters_to_compare` attribute.
        """
        retrieved_parameters = []
        with open("/app/Resources/other/polish_sentences.csv", "r", encoding='utf-8') as f:
            phrases = [line.strip() for line in f.readlines()]

        if self.iterations > len(phrases):
            info = "Number of iterations is smaller than the number of phrases in the CSV file. " + \
                f"Number of iterations was set to {self.iterations}"
            print(info)
            iterations = len(phrases)
        else:
            iterations = self.iterations

        for a in range(iterations):
            phrase = phrases[a]
            path = TMP_DIRECTORY_PATH + self.file_name_prefix + str(a) + ".wav"
            parameters = self.process_latent_and_embedding(self.embedding, self.latent, path, phrase)
            retrieved_parameters.append(parameters)
        self.parameters_to_compare = retrieved_parameters

    def _generate_histograms(self):
        """
        Generate and save histograms for each parameter in the comparison dictionary.
        This method iterates over the `parameters_to_compare_dict` attribute, calculates the mean and standard deviation
        for each parameter's values, and generates a histogram plot with a kernel density estimate (KDE) overlay. The plot
        is saved as a PNG file with a filename corresponding to the parameter key.
        """

        for key, values in self.parameters_to_compare_dict.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            title = f"{TITLE_PREFIX}; {key}, mean={mean_value:.6f}, std={std_value:.6f}"
            plt.figure(figsize=(10, 6))
            sns.histplot(data=values, bins='auto', kde=True, color=self.color)
            plt.title(title)
            plt.savefig(f"{SAVE_PATH}/{key}.png")
            plt.close()

    def run_comparison(self):
        """
        Run the comparison of parameters.
        This method retrieves the parameters to compare, converts the list of dictionaries
        to a dictionary of lists, and generates histograms for the comparison.
        Steps:
            1. Retrieve parameters to compare.
            2. Convert the list of dictionaries to a dictionary of lists.
            3. Generate histograms for the comparison.
        """

        self._retrieve_parameters_to_compare()
        self.parameters_to_compare_dict = self.convert_list_of_dicts_to_dict_of_lists(self.parameters_to_compare)
        self._generate_histograms()
