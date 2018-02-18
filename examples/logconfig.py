LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'simple': {
            'format': '{name:<12} {levelname:<8} {message}',
            'style': '{',
        },
    },

    # NOTE silly to filter by name....
    # 'filters': {
    #     'agents': {
    #         '()': 'logging.Filter',
    #         'name': 'rl.mdp.agents',
    #     },
    # },

    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'debug.log',
            'mode': 'w',
            'formatter': 'simple',
            # 'filters': ['agents'],
        },
    },

    'loggers': {
        'rl.pomdp': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
