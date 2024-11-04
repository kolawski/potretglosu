import os
from pydub import AudioSegment

txt_file_path = 'Resources/reference.txt'
source_folder = 'Resources/clips'
destination_folder = 'Resources/ready_audio_samples'

with open(txt_file_path, 'r') as file:
    mp3_files = [line.strip() for line in file]

for file_name in mp3_files:
    source_file = os.path.join(source_folder, file_name)
    destination_file = os.path.join(destination_folder, os.path.splitext(file_name)[0] + '.wav')

    if os.path.isfile(source_file):
        try:
            audio = AudioSegment.from_mp3(source_file)
            audio.export(destination_file, format="wav")
            print(f'Przekonwertowano: {file_name} na {destination_file}')
        except Exception as e:
            print(f'Błąd podczas konwersji {file_name}: {e}')
    else:
        print(f'Plik nie znaleziony: {file_name}')

