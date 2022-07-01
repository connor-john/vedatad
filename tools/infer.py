import argparse
import torch

from vedacore.misc import Config, load_weights
from vedatad.datasets.pipelines import Compose
from vedatad.engines import build_engine

BASE_META = {
    'video_info': {
        'video_name': 'video_test_0000131', 
        'duration': 130.63333333333333, 
        'frames': 3266, 
        'fps': 25, 
        'height': 128, 
        'width': 128, 
    },
    'video_prefix': 'data/thumos14/frames/test'
}

CLASSES = (
    'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
    'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
    'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
    'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
    'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
    'VolleyballSpiking'
)

def format_results(results):
    out_res = []
    video_name = BASE_META['video_info']['video_name']
    for label, segments in enumerate(results):
        for segment in segments:
            start, end, score = segment.tolist()
            label_name = CLASSES[label]
            res = dict(
                segment=[start, end], score=score, label=label_name
            )
            out_res.append(res)

    return out_res, video_name


def parse_args():
    parser = argparse.ArgumentParser(description='Infer activity')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('imgs', help='folder containing imgs')

    args = parser.parse_args()
    return args

def run(engine, pipeline, metas):

    data = pipeline(metas)
    engine.eval()

    data['video_metas'] = [[video_metas.data] for video_metas in data['video_metas']]
    data['imgs'] = [img.data.unsqueeze(0).cuda(torch.cuda.current_device()) for img in data['imgs']]

    with torch.no_grad():
        result = engine.infer(**data)[0]

    res, video = format_results(result)

    print(video)
    top = max(res, key=lambda x:x['score'])
    print('top')
    print(top)
    print('more')
    out = [item for item in res if item['score'] > 0.4]
    print(out)
    return result

def prepare(cfg, checkpoint):
    engine = build_engine(cfg.infer_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = engine.to(device)
    data_pipeline = Compose(cfg.data_pipeline)

    return engine, data_pipeline

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    engine, data_pipeline = prepare(cfg, args.checkpoint)

    video_meta = BASE_META.copy()
    video_meta['img_ids'] = list(range(BASE_META['video_info']['frames']))
    video_meta['ori_tsize'] = BASE_META['video_info']['frames']
    video_meta['tsize'] = BASE_META['video_info']['frames']
    video_meta['fps'] = BASE_META['video_info']['fps']
    video_meta['duration'] = BASE_META['video_info']['duration']

    run(engine, data_pipeline, video_meta)

if __name__ == '__main__':
    main()
