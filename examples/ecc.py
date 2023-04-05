import sys
import eccodes

def decode(fpath):
    r = []
    i = 0
    with open(fpath, "rb") as f:
        while True:
            msg = eccodes.codes_any_new_from_file(f)
            if msg is None:
                break
            
            i += 1
            print("message %d ----------------------------------", i)

            # print metadata key-values 
            it = eccodes.codes_keys_iterator_new(msg, 'ls')
            while eccodes.codes_keys_iterator_next(it):
                k = eccodes.codes_keys_iterator_get_name(it)
                v = eccodes.codes_get_string(msg, k)
                print("%s = %s" % (k, v))
            eccodes.codes_keys_iterator_delete(it)
            
            # get the lats, lons, values
            lats = eccodes.codes_get_double_array(msg, "latitudes")
            print(lats)
            lons = eccodes.codes_get_double_array(msg, "longitudes")
            print(lons)
            values = eccodes.codes_get_double_array(msg, "values")
            print(values)
            r.append([lats, lons, values])
            
            eccodes.codes_release(msg)
    
    f.close()
    return r

def main():
    try:
        values = decode(sys.argv[1])
        print(values)
    except eccodes.CodesInternalError as err:
        if eccodes.VERBOSE:
            eccodes.traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write(err.msg + '\n')

        return 1

if __name__ == "__main__":
    sys.exit(main())
