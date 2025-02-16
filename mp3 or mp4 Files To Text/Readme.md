# Whisper Transcription Project

This project uses OpenAI's Whisper model to transcribe audio and video files into text. The script searches for media files in a specified directory, transcribes them using the Whisper model, and saves the transcriptions as text files in the same directory as the original media files.

## Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

## Installation

1. Clone the repository or download the script files to your local machine.

2. Navigate to the project directory:
    ```sh
    cd "D:\Pyhton Projects"
    ```

3. Create a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    ```

4. Activate the virtual environment:
    - On Windows (Command Prompt):
        ```sh
        .\venv\Scripts\activate
        ```
    - On Windows (PowerShell):
        ```sh
        .\venv\Scripts\Activate.ps1
        ```

5. Install the required packages:
    ```sh
    pip install whisper
    ```

## Usage

1. Ensure your media files (audio/video) are located in the specified directory (`D:\Pyhton Projects`).

2. Run the transcription script:
    ```sh
    python Final.py
    ```

3. The script will search for media files with extensions `.mp3`, `.mp4`, and `.mkv` in the specified directory, transcribe them, and save the transcriptions as text files in the same directory as the original media files.

## Script Details

- `searchfile()`: Searches for media files in the specified directory and returns a list of file paths.
- `scriptwriter(media_file)`: Transcribes the given media file using the Whisper model and saves the transcription as a text file in the same directory.
- `main()`: Main function that orchestrates the search and transcription process.

## Example

If you have a media file named `example.mp3` in the `D:\Pyhton Projects` directory, the script will create a transcription file named `example.txt` in the same directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) - The transcription model used in this project.
