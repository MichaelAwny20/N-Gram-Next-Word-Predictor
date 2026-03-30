import os
from pathlib import Path
from dotenv import load_dotenv

class Normalizer:
    """
    Responsible for loading, cleaning, tokenizing, and saving the corpus.
    This class ensures the training data and user input are processed identically.
    """
    def __init__(self):
        """Initializes the Normalizer."""
        pass
    def strip_gutenberg(self, text: str) -> str:
        """
        Removes the Project Gutenberg header and footer.
        
        :param text: The full raw text of the book.
        :return: The text containing only the story content.
        """
        # 1. Define the markers (exactly as they appear in the files)
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

        # 2. Find the position of the start marker
        start_index = text.find(start_marker)
        if start_index != -1:
            # We want to start AFTER the marker, so find the end of that line
            # Look for the next newline character '\n' after the marker
            start_index = text.find("\n", start_index) + 1
        else:
            # If marker not found, start from the beginning
            start_index = 0

        # 3. Find the position of the end marker
        end_index = text.find(end_marker)
        if end_index == -1:
            # If marker not found, go to the very end
            end_index = len(text)

        # 4. Slice the text: Keep everything from start_index to end_index
        return text[start_index:end_index].strip()
    def load(self, folder_path: str) -> str:
        """
        Loads all .txt files from a specific folder and merges them.
        :param folder_path: String path to the folder containing raw .txt files.
        :return: A single string containing the combined text of all files.
        """
        combined_text = []
        # Convert string path to a Path object for easier navigation
        path = Path(folder_path)
        
        # Check if the folder actually exists to prevent crashes
        if not path.exists():
            print(f"Error: The folder {folder_path} does not exist.")
            return ""
        files_found_count = 0
        # Loop through every file in the folder
        for file_path in path.iterdir():
            # Only process files that end in .txt
            if file_path.suffix == '.txt':
                print(f"Loading: {file_path.name}")
                files_found_count += 1
                # 'utf-8' encoding ensures special characters (like curly quotes) load correctly
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                    
                    # --- THE FIX: Clean THIS specific file now ---
                    clean_content = self.strip_gutenberg(raw_content)
                    
                    combined_text.append(clean_content)
        if files_found_count == 0:
            print(f"Warning: No .txt files were found in '{folder_path}'.")
            return ""            

        # Join all the book strings into one giant string with a space between them
        return " ".join(combined_text)
    def lowercase(self, text: str) -> str:
        """Converts all characters to lowercase."""
        return text.lower()
    def remove_punctuation(self, text: str) -> str:
        """Removes all punctuation using Regex."""
        import re
        # This looks for anything that ISN'T a word character or a space
        return re.sub(r'[^\w\s]|_', '', text)
    def remove_numbers(self, text: str) -> str:
        """Removes all digits from the text."""
        import re
        return re.sub(r'\d+', '', text)
    def remove_whitespace(self, text: str) -> str:
        """
        Removes extra spaces and tabs, but preserves the original 
        line-by-line structure.
        """
        # 1. Break the text into a list of lines using the line breaks (\n)
        lines = text.splitlines()
        
        cleaned_lines = []
        for line in lines:
            # 2. For each individual line, split by space/tab 
            # and join with a single space. This removes the 'junk' gaps.
            clean_line = " ".join(line.split())
            
            # 3. Only add the line if it isn't empty
            if clean_line:
                cleaned_lines.append(clean_line)
        
        # 4. Join the cleaned lines back together using a newline
        return "\n".join(cleaned_lines)
    def normalize(self, text: str) -> str:

        """
        The main entry point for cleaning. 
        Applies lowercase -> punctuation -> numbers -> whitespace.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text
    def sentence_tokenize(self, text: str) -> list[str]:
        """
        Splits the raw text into a list of sentences.
        Each sentence is identified by a terminal punctuation mark (. ! or ?).
        
        :param text: The raw (but stripped of Gutenberg headers) book text.
        :return: A list where each element is one sentence.
        """
        import re
        
        # We look for . ! or ? that is followed by a space or a new line
        # The (?<=[.!?]) is a "lookbehind" - it keeps the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # We use a "List Comprehension" to remove any empty sentences
        return [s.strip() for s in sentences if s.strip()]
    def word_tokenize(self, sentence: str) -> list[str]:
        """
        Splits a single clean sentence into a list of words.
        """
        # .split() automatically splits by any whitespace
        return sentence.split()  
    def process_all(self, raw_text: str) -> list[list[str]]:
        """
        The "Master Loop" that connects all your steps.
        1. Sentence Tokenize (Save the fences)
        2. Loop through each sentence:
           - Normalize (Clean the sentence)
           - Word Tokenize (Chop into words)
        """
        final_corpus = []

        # Step 1: Split into sentences first!
        sentences = self.sentence_tokenize(raw_text)

        # Step 2: The Loop
        for sentence in sentences:
            # Clean it (lowercase, remove punct, remove numbers)
            clean_sentence = self.normalize(sentence)
            
            # Chop it into a list of words
            words = self.word_tokenize(clean_sentence)
            
            # Only add it if the sentence isn't empty after cleaning
            if len(words) > 0:
                final_corpus.append(words)

        return final_corpus
    def save(self, sentences: list[list[str]], filepath: str):
        """
        Writes the tokenized sentences to a file.
        Each sentence (list of words) becomes one line in the file.
        """
        # Ensure the directory exists before saving
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence_list in sentences:
                # Turn ['it', 'was'] into "it was"
                line = " ".join(sentence_list)
                # Write the line and add a 'newline' jump to the next line
                f.write(line + "\n")
        
        print(f"Successfully saved {len(sentences)} sentences to {filepath}")
# --- Internal Module Test ---
if __name__ == "__main__":
    # 1. Load the settings from your config folder
    load_dotenv('config/.env')
    
    # 2. Get the path from the .env file instead of typing it manually
    test_folder = os.getenv("TRAIN_RAW_DIR")

    # 3. Test the load method
    norm = Normalizer()
    result = norm.load(test_folder)
    result = norm.process_all(result)
    output_folder=os.getenv("TRAIN_TOKENS")
    print(f"Processed Result: {result}")
    norm.save(result,output_folder)

  
