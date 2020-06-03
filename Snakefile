import os.path
import sys
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Iterator, List, Union

RAW_DETECTION_ROOT = '/media/viki/DATA/Jian/epic-2020-hand-bboxes'
DATA_INTERIM = 'data/interim'
DATA_PROCESSED = 'data/processed'


def iter_video_dirs(root_dir: Union[Path, str]) -> Iterator[Path]:
    root_dir = Path(root_dir)
    def is_person_dir(p: Path) -> bool:
        return re.match('P\d+', p.name) is not None

    def is_video_dir(p: Path) -> bool:
        return re.match('P\d+_\d+', p.name) is not None

    for person_dir in filter(is_person_dir, root_dir.iterdir()):
        for video_dir in filter(is_video_dir, person_dir.iterdir()):
            yield video_dir

videos: List[str] = [
    video_dir.name
    for video_dir in iter_video_dirs(RAW_DETECTION_ROOT)
]

def extract_ids(video_name: str) -> Dict[str, str]:
    matches = re.match(r'P(\d+)_(\d+)', video_name)
    return {
        'person': f'P{matches.group(1)}',
        'video': f'P{matches.group(1)}_{matches.group(2)}',
    }

video_lengths = pd.read_csv('EPIC_100_frame_counts.csv', index_col='video_id')['rgb_n_frames']


def get_n_frames_for_video(video_id: str) -> int:
    return video_lengths.loc[video_id]


rule all:
    input: [f'{DATA_PROCESSED}/{ids["person"]}/{ids["video"]}.pkl' \
            for ids in map(extract_ids, videos)]


rule aggregate_frame_detections:
    input: RAW_DETECTION_ROOT + '/{person_id}/{video_id}/'
    output: DATA_INTERIM + '/{person_id}/{video_id}.pkl'
    shell:
        """
        python src/scripts/aggregate_raw_detections.py {input} {output}
        """

rule convert_raw_detections_to_releasable_detections:
    input: DATA_INTERIM + '/{person_id}/{video_id}.pkl'
    output: DATA_PROCESSED + '/{person_id}/{video_id}.pkl'
    shell:
        """
        python src/scripts/convert_raw_to_releasable_detections.py {input} {output}
        """

rule check_video_detections:
    input: DATA_PROCESSED + '/{person_id}/{video_id}.pkl'
    output: DATA_PROCESSED + '/{person_id}/.{video_id}.check'
    params:
        n_frames=lambda wildcards: get_n_frames_for_video(wildcards.video_id)
    shell:
         """
         python src/scripts/check_data.py {input} --n-frames {params.n_frames} && touch {output}
         """

rule checks:
    input: [f'{DATA_PROCESSED}/{ids["person"]}/.{ids["video"]}.check' \
            for ids in map(extract_ids, videos)]
