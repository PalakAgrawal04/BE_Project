"""
Voice query processing service using Whisper API.
"""

import os
import logging
from typing import Tuple, Optional
import tempfile
import base64
from datetime import datetime

import whisper
from pydub import AudioSegment
import numpy as np

logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self, model_name: str = "base"):
        """
        Initialize voice processor with Whisper model.
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium")
        """
        self.model = whisper.load_model(model_name)
        
    def process_audio(self, 
                     audio_data: str,
                     format: str = "wav") -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process base64 encoded audio to text.
        
        Args:
            audio_data: Base64 encoded audio data
            format: Audio format (wav, mp3, etc.)
            
        Returns:
            Tuple of (success, transcribed_text, error_message)
        """
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{format}") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()
                
                # Load and preprocess audio
                audio = AudioSegment.from_file(temp_audio.name, format=format)
                
                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples())
                
                # Transcribe with Whisper
                result = self.model.transcribe(samples)
                transcribed_text = result["text"].strip()
                
                if not transcribed_text:
                    return False, None, "No speech detected in audio"
                
                return True, transcribed_text, None
                
        except Exception as e:
            error_msg = f"Voice processing failed: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg