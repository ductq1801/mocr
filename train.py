import argparse

from trainer import Trainer
from config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')
    parser.add_argument('--pretrained',required=False,default=False)
    parser.add_argument('--debug',required=False,default=False)
    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    trainer = Trainer(config,pretrained=args.pretrained,debug=args.debug)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    trainer.train()

if __name__ == '__main__':
    main()