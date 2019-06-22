def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('-e','--episodes', type=int, default=32, help='# of episodes')
    parser.add_argument('-lr','--lr', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('-lm','--load_model', type=str, help='model for testing')
    parser.add_argument('-ga','--gamma', type=float, default=0.99, help='discount rate for rewards')
    parser.add_argument("-mn","--model_name", type=str, default="model", help="model name for saving")
    return parser
