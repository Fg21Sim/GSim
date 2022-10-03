# Copyright (c) 2017-2022 Chenxi SHAN <cxshan@hey.com>
# General purpose log class

import logging
import time
from datetime import datetime
import pprint
import numpy as np
import sys
import getpass

class Log:
    r"""
        *** A general purpose log class *** # Utility
        !!! Use the loggor @ Property !!! # Notes
        +++  +++ # Todo
        :Params level_: NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL;
        :Params name: The name of your loggor;
    """
    def __init__( self, name=None, level_='debug', stdout_=False ):
        self.level_ = level_
        self.name = name
        self.stdout_ = stdout_
    
    @property
    def user( self ):
        return getpass.getuser()
    
    @property
    def logger( self ):
        logger = self.createlog()
        return logger
    
    def createlog( self ):
        r"""
            *** Create a logger object w/ intended level_ & name *** # Basic function
            !!!  !!! # Important note
            +++ Add multi-log support, such as multiple name :) +++ # Improvements
            #formatter = logging.Formatter
            ('%(asctime)s : %(name)s  : %(funcName)s : %%! (levelname)s : %(message)s') 
            # The name is a duplicate info I remove it here;
            :Params level_: NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL;
            :Params name: The name of your loggor;
            :Output loggor: a loggor object from python logging module
        """
        # Create time stamp, log file name, & initialize the data;
        d = datetime.now()
        dd = d.strftime('%m_%d_%Y_%H%M%S')
        logname = self.name + dd
        logfilename = self.name + '_' + dd + '.log'
        logger = logging.getLogger(logname)
        formatter = logging.Formatter('%(asctime)s : %(funcName)s : %(levelname)s : %(message)s')
        
        # File handler
        fileHandler = logging.FileHandler( logfilename )
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        # Stream handler
        if self.stdout_:
            consoleHandler = logging.StreamHandler(sys.stdout)
        else:
            consoleHandler = logging.StreamHandler()
        logger.addHandler(consoleHandler)
        consoleHandler.setFormatter(formatter)
        
        level_ = self.level_
        if level_ == 'debug':
            logger.setLevel(logging.DEBUG)
            fileHandler.setLevel(logging.DEBUG)
            consoleHandler.setLevel(logging.DEBUG)
        elif level_ == 'info':
            logger.setLevel(logging.INFO)
            fileHandler.setLevel(logging.INFO)
            consoleHandler.setLevel(logging.INFO)
        elif level_ == 'warning':
            logger.setLevel(logging.WARNING)
            fileHandler.setLevel(logging.WARNING)
            consoleHandler.setLevel(logging.WARNING)
        elif level_ == 'error':
            logger.setLevel(logging.ERROR)
            fileHandler.setLevel(logging.ERROR)
            consoleHandler.setLevel(logging.ERROR)
        elif level_ == 'critical':
            logger.setLevel(logging.CRITICAL)
            fileHandler.setLevel(logging.CRITICAL)
            consoleHandler.setLevel(logging.CRITICAL)
        else:
            logger.setLevel(logging.NOTSET)
            fileHandler.setLevel(logging.NOTSET)
            consoleHandler.setLevel(logging.NOTSET)
        
        # Intial logger info
        logger.info( "Log Header >> Action Time : Fuction or method: level : Message" )
        logger.info( "Log standard output streaming is %s" % self.stdout_ )
        logger.info( "%s log file is created" % logfilename )
        logger.info( "Log is created by %s" % self.user )
        logger.info( "Log level is %s" % level_ )
        return logger
    

class takelogger:
    r"""
        *** A fake class to take a loggor *** # Utility
        :Params logger_: The logger object;
    """
    def __init__( self, logger_=None ):
        self.loggor = logger_
        self.loggor.info('Initialization is done!')
    
    def change( self ):
        self.loggor.info('Somthing is changed!')
        
    def bla( self ):
        self.loggor.info('Bla, bla, bla')
        

def demo_logger():
    r"""
        *** Demo a loggor *** # Utility
        It calls the Log class & takelogger class
    """
    loging = Log('Demo','debug')
    Test = takelogger(loging.logger)
    Test.change()
    Test.bla()
    Test.change()
    Test.bla()
    print('If you see the file created & the demo info, you are all good!')