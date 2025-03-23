import logging

class Logger:
    def __init__(
            self, 
            log_file="app.log", 
            file_level=logging.INFO, 
            console_level=logging.INFO
        ):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg):
        self.logger.info(msg)

    @staticmethod
    def shutdown():
        logging.shutdown()


logger = Logger()
