import tyro

from rr_gs_test.api.view_splat_with_cams import ViewSplatConfig, main

if __name__ == "__main__":
    main(tyro.cli(ViewSplatConfig))
