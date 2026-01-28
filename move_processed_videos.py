import os
import shutil

video_dir = "/Users/shantanuchandra/Downloads/Logging_PhoropterUI/Sample/videos"
csv_dir = "/Users/shantanuchandra/Downloads/Logging_PhoropterUI/MatchedScreens"
dest_dir = "/Users/shantanuchandra/Downloads/Logging_PhoropterUI/Sample/videos-done"

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
csvs = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

csv_basenames = {os.path.splitext(f)[0] for f in csvs}

moved_count = 0
for video in videos:
    video_basename = os.path.splitext(video)[0]
    if video_basename in csv_basenames:
        src = os.path.join(video_dir, video)
        dst = os.path.join(dest_dir, video)
        print(f"Moving {video} to {dest_dir}")
        shutil.move(src, dst)
        moved_count += 1

print(f"Total videos moved: {moved_count}")
