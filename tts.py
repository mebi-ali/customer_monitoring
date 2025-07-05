
# import asyncio
# import edge_tts
# import tempfile
# import os
# from playsound import playsound

# class TextToSpeech:
#     async def speak_async(self, text):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
#             file_path = fp.name

#         communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
#         await communicate.save(file_path)
#         playsound(file_path)
#         os.remove(file_path)

#     def speak(self, text):
#         asyncio.run(self.speak_async(text))


# import asyncio
# import edge_tts
# import tempfile
# import os
# import subprocess

# class TextToSpeech:
#     async def speak_async(self, text):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
#             file_path = fp.name

#         # Generate TTS audio
#         communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
#         await communicate.save(file_path)

#         # Play audio in a separate subprocess (non-blocking)
#         try:
#             if os.name == "nt":
#                 # Windows
#                 subprocess.Popen(["start", "/min", file_path], shell=True)
#             elif os.uname().sysname == "Darwin":
#                 # macOS
#                 subprocess.Popen(["afplay", file_path])
#             else:
#                 # Linux
#                 subprocess.Popen(["mpg123", "-q", file_path])
#         except Exception as e:
#             print(f"Audio playback failed: {e}")
#             os.remove(file_path)
#             return

#         # Optionally: delay removal or use a watcher to delete later
#         await asyncio.sleep(8)  # Adjust based on text length
#         os.remove(file_path)

#     def speak(self, text):
#         try:
#             asyncio.run(self.speak_async(text))
#         except KeyboardInterrupt:
#             print("\n[Interrupted] Speech was canceled.")

import asyncio
import edge_tts
import tempfile
import os
import subprocess
import platform # To correctly identify OS for playback

class TextToSpeech:
    async def speak_async(self, text):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            file_path = fp.name

        print(f"Generating TTS audio to: {file_path}")
        # Generate TTS audio
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        await communicate.save(file_path)
        print("TTS audio generation complete.")

        # Play audio and wait for it to finish
        try:
            player_command = []
            current_os = platform.system()

            if current_os == "Windows":
                # Use 'start' with '/wait' for blocking behavior on Windows.
                # 'shell=True' is required for 'start' command to work.
                player_command = ["start", "/wait", file_path]
                print(f"Playing audio on Windows: {player_command}")
                subprocess.run(player_command, shell=True, check=True)
            elif current_os == "Darwin":
                # macOS
                player_command = ["afplay", file_path]
                print(f"Playing audio on macOS: {player_command}")
                subprocess.run(player_command, check=True)
            else:
                # Linux (requires mpg123, paplay, or similar)
                # Try mpg123 first, then paplay if mpg123 is not found.
                try:
                    player_command = ["mpg123", "-q", file_path]
                    print(f"Playing audio on Linux (mpg123): {player_command}")
                    subprocess.run(player_command, check=True)
                except FileNotFoundError:
                    try:
                        player_command = ["paplay", file_path]
                        print(f"Playing audio on Linux (paplay): {player_command}")
                        subprocess.run(player_command, check=True)
                    except FileNotFoundError:
                        print("Error: Neither mpg123 nor paplay found. Please install one of them (e.g., sudo apt install mpg123).")
                        return

        except FileNotFoundError as fnfe:
            print(f"Audio playback command not found: {fnfe}. Please ensure the audio player is installed and in your PATH.")
        except subprocess.CalledProcessError as cpe:
            print(f"Audio playback process failed with exit code {cpe.returncode}. Stderr: {cpe.stderr.decode() if cpe.stderr else 'N/A'}")
        except Exception as e:
            print(f"Audio playback failed unexpectedly: {e}")
        finally:
            # Clean up the temporary file after playback (or error)
            if os.path.exists(file_path):
                print(f"Deleting temporary audio file: {file_path}")
                os.remove(file_path)

    def speak(self, text):
        try:
            asyncio.run(self.speak_async(text))
        except KeyboardInterrupt:
            print("\n[Interrupted] Speech was canceled.")
        except Exception as e:
            print(f"Error in TextToSpeech.speak: {e}")