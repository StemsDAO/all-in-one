from pydantic import BaseModel

class GetAudioFeaturesBaseRequest(BaseModel):
    mix_path: str
    bass_path: str
    drums_path: str
    music_path: str
    vocals_path: str
