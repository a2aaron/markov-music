set -e
cargo run
fluidsynth -F output_sound.wav megalo.sf2 out.mid
rm output_sound.mp3
ffmpeg -i output_sound.wav -vn -ar 44100 -ac 2 -b:a 192k output_sound.mp3