import argparse


def parseArgumets():
    parser = argparse.ArgumentParser(description="Script for training and plotting")
    
    parser.add_argument('-train', type=str, default='True', 
                    help='Set to False to skip training')
    
    parser.add_argument('-config_path', type=str, default='Adam', 
                        help='path to the config file')

    args = parser.parse_args()

    config = {}
    config['do training'] = args.train.lower() in ('true', '1', 't', 'yes')
    config['config_path'] = args.config_path


    return config