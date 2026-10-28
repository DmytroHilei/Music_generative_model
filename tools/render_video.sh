#!/bin/bash
# MP3 -> MP4 with a constant-Q spectrum (ffmpeg showcqt) and a caption line, for sharing generated samples.
# Usage: tools/render_video.sh in.mp3 out.mp4 "caption text" [seconds]
in=$1; out=$2; text=$3; secs=${4:-60}
font=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
esc=$(printf '%s' "$text" | sed -e "s/\\\\/\\\\\\\\/g" -e "s/:/\\\\:/g" -e "s/'/\\\\'/g" -e "s/%/\\\\%/g")
ffmpeg -y -loglevel error -t "$secs" -i "$in" -filter_complex \
  "[0:a]showcqt=s=1280x720:fps=30:bar_g=2:sono_g=4:axis=0:count=3:basefreq=40:endfreq=8000,format=yuv420p,drawbox=x=0:y=0:w=iw:h=86:color=black@0.6:t=fill,drawtext=fontfile=$font:text='$esc':fontcolor=white:fontsize=30:x=(w-text_w)/2:y=28[v]" \
  -map "[v]" -map 0:a -c:v libx264 -preset veryfast -crf 23 -c:a aac -b:a 160k -shortest -movflags +faststart "$out"
