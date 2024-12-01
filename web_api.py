from pathlib import Path
from fastapi import FastAPI
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
import requests
from contextlib import ExitStack

from web.app_types import GetAudioFeaturesBaseRequest
from web.utils import TimeTrack
from src.allin1.analyze import analyze
import uuid

app = FastAPI(title="AllIn1 Audio Features", description="extract audio features such as rhythm, beats positions, etc from pre-processed stem files", version="1.0")
path = "/dev/shm"

toDelete = True

def get_temp_file_downloaded(ctx, dir, file_name, url):
    file_path = dir / Path(file_name)
    file = open(file_path, "wb")
    file_response = requests.get(url)
    file.write(file_response.content)
    file.flush()
    return Path(file.name)

def upload_to_s3(file_name, bucket_name, object_name):
    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        return True
    except ClientError as e:
        print(e)
        return False

@app.post("/get-audio-features", tags=['audio_features'])
async def get_audio_features(r: GetAudioFeaturesBaseRequest):
    audio_features = {}

    print('Start Loading: ' + r.mix_path)

    # Download files
    with TemporaryDirectory(dir=path, prefix="stems_demucs_"+str(uuid.uuid4())) as temp_dir:
        with ExitStack() as stack:
            mix_file_path = get_temp_file_downloaded(ctx=stack, dir=temp_dir, file_name="mix.wav", url=r.mix_path)
            bass_file_path = get_temp_file_downloaded(ctx=stack, dir=temp_dir, file_name="bass.wav", url=r.bass_path)
            drums_file_path = get_temp_file_downloaded(ctx=stack, dir=temp_dir, file_name="drums.wav", url=r.drums_path)
            music_file_path = get_temp_file_downloaded(ctx=stack, dir=temp_dir, file_name="other.wav", url=r.music_path)
            vocals_file_path = get_temp_file_downloaded(ctx=stack, dir=temp_dir, file_name="vocals.wav", url=r.vocals_path)
            # demucs_paths = [bass_file_path, drums_file_path, music_file_path, vocals_file_path]
            demucs_paths = [Path(temp_dir)]

            with TimeTrack("allin1.rhythm"):
                data = analyze(
                    paths=str(mix_file_path),
                    demucs_paths=demucs_paths,
                    visualize=False,
                    sonify=False,
                    device='cuda',
                    include_activations=False,
                    include_embeddings=False,
                    keep_byproducts=False,
                    overwrite=False,
                    multiprocess=True,
                    )

                audio_features["bpm"] = data.bpm
                audio_features["beats"] = data.beats
                audio_features["beat_positions"] = data.beat_positions
                audio_features["downbeats"] = data.downbeats
                audio_features["segments"] = data.segments

            return audio_features

