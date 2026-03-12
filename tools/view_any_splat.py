import tyro

from rr_gs_test.api.view_any_splat import ViewSplatConfig, main

if __name__ == "__main__":
    main(tyro.cli(ViewSplatConfig))
