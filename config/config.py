#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 服务配置管理

import configparser


class Config:

    def __init__(self):
        # 加载配置
        conf = configparser.ConfigParser()
        conf.read('./config/default.ini')
        self.conf = conf


config = Config().conf
