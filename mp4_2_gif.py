import os
import moviepy.editor as mpy


for task in range(1, 6):
    path = './log/gif'
    filename = f'DQN_{task}.mp4'
    # task = filename[4]

    filepath = os.path.join(path, filename)
    gif_filepath = os.path.join(path, filename.replace('.mp4', '.gif'))

    video = mpy.VideoFileClip(filepath)
    video = video.resize((320, 240))
    video.write_gif(gif_filepath, fps=10)

