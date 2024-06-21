import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='-----VDNLL --PyTorch ')
    
    # Architecture
    parser.add_argument('--arch', default="PL", choices=["PL", "Co_Teaching", "Co_Teaching_plus", "Decoupling"])
    parser.add_argument('--feature_model', default="TextCNN", choices=["LSTM", "Transformer"])
    parser.add_argument('--embedding_dim', default=256, type=int)
    parser.add_argument('--vocab_size', default=50000, type=int)
    
    # loader
    parser.add_argument('--is_balanced', default=True)
    
    # Data
    parser.add_argument('--SC_Type', default="RE", choices=["RE", "TD", "IOU"])
    parser.add_argument('--num_classes', type=int, default="2", help='number of class')
    parser.add_argument('--max_setence_length', type=int, default="2000", help='Max length of a setence')
    parser.add_argument('--training_data_ratio', default=0.80, type=float)
    parser.add_argument('--valid_data_ratio', default=0.10, type=float)
    parser.add_argument('--test_data_ratio', default=0.10, type=float)
    
    parser.add_argument('--batch_size', default=32, type=int, help='batch size (default: 32)')
    parser.add_argument('--mislabel_rate', default=0.20, type=float)
    parser.add_argument('--forget_rate', default=0.20, type=float)
    
    # Optimization
    parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout (default: 0)')
    parser.add_argument('--epochs', type=int, default="100", help='number of total training epochs')
    parser.add_argument('--optim', default="adamw", type=str, metavar='TYPE', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')

    # Log and save
    parser.add_argument('--log_dir', default='./Logs', type=str, metavar='DIR')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, metavar='DIR')


    return parser.parse_args()
