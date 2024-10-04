import logging, time, os


def get_log_path(args):
    # 创建日志文件
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    
    new_log_dir = os.path.join(args.log_dir, args.arch)
    if not os.path.exists(new_log_dir):
        os.mkdir(new_log_dir)
    
    # 获取当前时间作为日志文件名称
    current_time=time.strftime('%Y%m%d%H%M',time.localtime(time.time() )) 
    
    if args.mislabel_rate > 0:
        mislable_rate = 100 * args.mislabel_rate
        log_txt_name = args.model + "_noise_" + str(mislable_rate) + "_" + args.SC_Type + "_" + current_time + ".txt"
    else:
        log_txt_name = args.model + "_" + args.SC_Type + "_" + current_time + ".txt"
    log_txt_path = os.path.join(new_log_dir, log_txt_name)
    return log_txt_path, log_txt_name


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger