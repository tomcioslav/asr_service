from pydub import AudioSegment

def convert_to_wav(file_path, target_sr=44100, output_path=None):
    # Load audio file (supports both webm and wav)
    audio = AudioSegment.from_file(file_path)
    
    # Convert to wav format in memory
    audio = audio.set_frame_rate(target_sr)
    
    
    if output_path is None:
        # Use the same filename but with .wav extension
        output_path = file_path.with_suffix('.wav')
    audio.export(output_path, format="wav")
    return output_path