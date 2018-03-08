LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'simple': {
            'format': '{levelname:<8} {name:<12} \t {message}',
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
        'rl.pomdp.policies': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
        # 'rl.pomdp.agents.pgradient': {
        #     'handlers': ['file'],
        #     'level': 'INFO',
        #     'propagate': True,
        # },
    },
}
