import weakref


def to_dead(wref):
    '''
    returns true if object referred to by weak reference is dead
    '''
    #len([u for u in str(_11).replace('>',' ').split(' ') if u == 'dead'])
    return any([u=='dead' for u in str(wref).replace('>',' ').split(' ')])


def get_mmaps():
     '''
     gets filename for each memory-mapped variable
    Usage
    >>> mmaps = get_mmaps()
    >>> for k,v in mmaps.items():
          print(k,v)
    '''
    mempaths={}
    globals_copy = globals().copy()
    for name in globals_copy.keys():
        try:
            name_ref = weakref.ref(eval(compile(name,'tmp.txt', 'eval')))
        except (NameError, TypeError):
            continue
        if hasattr(name_ref(), '__dict__'):
            v = vars(name_ref())
            if '_mmap' in v.keys():
                mempaths[name] = v['filename']

    return mempaths



def get_weakrefs():
    '''
    Usage
    >>> weakrefs = get_weakrefs()
    >>> for k,v in weakrefs.items():
          print(k,v)
    '''

    weak={}

    globals_copy = globals().copy()
    for f in globals_copy.keys():
        if type(globals_copy[f])==weakref.ref:
            weak[f]=globals_copy[f]
    return weak


if __name__ == '__main__':
  
  print('\n mmaps:  \n')
  
  mmaps = get_mmaps()

  for k,v in mmaps.items():
    print(k,v)

  print('\n Weak References:  \n')

  weakrefs = get_weakrefs()
  for k,v in weakrefs.items():
        print(k,v)