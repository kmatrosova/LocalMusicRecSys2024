from recbole.quick_start import run_recbole

if __name__ == '__main__':
    for platform in ['LFM', 'DEEZER']:
        for country in ['GLOBAL', 'FR', 'DE', 'BR']:
            for model in ['ItemKNN', 'NeuMF']:
                run_recbole(model=model, dataset=platform+'_'+country, config_dict={'field_separator': ',',
                                                                                    'platform': platform,
                                                                                    'country': country})
