from easydict import EasyDict


def print_edict(edict: EasyDict, __prefix=''):
    if not __prefix:
        print(f'# EasyDict object id {id(edict)}')
    for k, v in edict.items():
        if isinstance(v, EasyDict):
            print(f'\n{__prefix}{k}:')
            print_edict(v, __prefix=f'  {__prefix}')
        else:
            print(f'{__prefix}{k}: {v}')
    # if not __prefix:
    #     print()


def flatten_edict(edict: EasyDict, sep='/', __prefix=''):
    flat_edict = EasyDict()
    for k, v in edict.items():
        new_key = f'{__prefix}{sep if __prefix else ""}{k}'
        if isinstance(v, EasyDict):
            flat_edict.update(flatten_edict(v, __prefix=new_key))
        else:
            flat_edict[new_key] = v
    return flat_edict


if __name__ == '__main__':
    edict = EasyDict(
        a=EasyDict(
            one=1,
            two=2,
            three=3
        ),
        b=EasyDict(
            one=EasyDict(
                i=EasyDict(
                    yup='y um',
                    num='n up'
                )
            )
        )
    )

    print(edict)
    print('=' * 100)
    print_edict(edict)
    print('=' * 100)
    print_edict(flatten_edict(edict))
