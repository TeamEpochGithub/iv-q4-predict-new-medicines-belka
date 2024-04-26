# Run mp3 file

file_name = "notebooks/jesse.mp3"

import pygame
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(file_name)
pygame.mixer.music.play()


if __name__ == "__main__":
    print("Playing mp3 file")
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    print("Done playing mp3 file")
    pygame.quit()