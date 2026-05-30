import os
import shutil
import subprocess
import tempfile
from pathlib import Path

GENERATOR_DIR = Path(__file__).resolve().parent / 'polymatrix-generators-master'
PMGAMES_DIR = Path(__file__).resolve().parent / 'PMGames'
PMGEN_BINARY = GENERATOR_DIR / 'pm-gen'

GAME_SPECS = [
    # (generator name, graph type, actions, g1, g2, p)
    ('CoordZero', 'Complete', 4, 4, 1, 0.5),
    ('GroupZero', 'Cycle', 3, 5, 1, 2.0),
    ('StrictComp', 'Grid', 3, 2, 3, 0.0),
    ('WeightCoop', 'Complete', 3, 4, 1, 2.0),
    ('CoordZero', 'Tree', 4, 4, 1, 0.5),
    ('GroupZero', 'Complete', 3, 5, 1, 2.0),
    ('StrictComp', 'Tree', 3, 2, 3, 0.0),
    ('WeightCoop', 'Cycle', 3, 4, 1, 2.0),
    ('CoordZero', 'Cycle', 4, 4, 1, 0.5),
    ('GroupZero', 'Grid', 3, 5, 1, 2.0),
    ('StrictComp', 'Cycle', 3, 2, 3, 0.0),
    ('WeightCoop', 'Tree', 3, 4, 1, 2.0),
    ('CoordZero', 'Grid', 4, 4, 1, 0.5),
    ('GroupZero', 'Tree', 3, 5, 1, 2.0),
    ('StrictComp', 'Complete', 3, 2, 3, 0.0),
    ('WeightCoop', 'Grid', 3, 4, 1, 2.0)
    
]

GMP_INCLUDE_PATHS = [
    os.environ.get('CONDA_PREFIX', '/opt/anaconda3') + '/include',
    '/usr/local/include',
    '/usr/include',
]
GMP_LIBRARY_PATHS = [
    os.environ.get('CONDA_PREFIX', '/opt/anaconda3') + '/lib',
    '/usr/local/lib',
    '/usr/lib',
]


def ensure_pm_gen():
    if PMGEN_BINARY.exists() and os.access(PMGEN_BINARY, os.X_OK):
        PMGEN_BINARY.unlink()

    print('Compiling polymatrix generator binary using direct compiler...')
    cc = shutil.which('cc') or shutil.which('clang')
    if cc is None:
        raise RuntimeError('No C compiler found on PATH.')

    include_dir = None
    for path in GMP_INCLUDE_PATHS:
        if Path(path, 'gmp.h').exists():
            include_dir = path
            break

    lib_dir = None
    for path in GMP_LIBRARY_PATHS:
        if any(Path(path, lib).exists() for lib in ('libgmp.dylib', 'libgmp.a', 'libgmp.so')):
            lib_dir = path
            break

    if include_dir is None or lib_dir is None:
        raise RuntimeError(
            'GMP header or library not found. Install GMP or activate a conda environment with gmp.'
        )

    compile_cmd = [
        cc,
        '-c',
        '-g',
        '-Wall',
        'pm-gen.c',
        'coord_zero.c',
        'strict_comp.c',
        'weight_coop.c',
        'bayesian.c',
        'bimatrix.c',
        'polymatrix.c',
        'graph.c',
        'matrix.c',
        'util.c',
        '-I.',
        '-I' + include_dir,
    ]

    link_cmd = [
        cc,
        '-g',
        '-Wall',
        'pm-gen.o',
        'coord_zero.o',
        'strict_comp.o',
        'weight_coop.o',
        'bayesian.o',
        'bimatrix.o',
        'polymatrix.o',
        'graph.o',
        'matrix.o',
        'util.o',
        '-o',
        str(PMGEN_BINARY),
        '-L' + lib_dir,
        '-Wl,-rpath,' + lib_dir,
        '-lgmp',
        '-lm',
    ]

    old_cwd = os.getcwd()
    os.chdir(GENERATOR_DIR)
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError('Compilation failed:\n' + result.stdout + result.stderr)

        result = subprocess.run(link_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError('Linking failed:\n' + result.stdout + result.stderr)

    finally:
        for obj in ('pm-gen.o', 'coord_zero.o', 'strict_comp.o', 'weight_coop.o', 'polymatrix.o', 'graph.o', 'matrix.o', 'util.o'):
            try:
                os.remove(GENERATOR_DIR / obj)
            except OSError:
                pass
        os.chdir(old_cwd)

    if not PMGEN_BINARY.exists() or not os.access(PMGEN_BINARY, os.X_OK):
        raise RuntimeError('pm-gen binary was not created successfully.')


TARGET_PAYOFF_MAX = 100


def parse_pm_gen_file(path):
    with open(path, 'r') as f:
        tokens = f.read().split()

    if len(tokens) < 1:
        raise ValueError('Empty generator output file: %s' % path)

    players = int(tokens[0])
    graph_values = []
    expected_graph_values = players * players
    if len(tokens) < 1 + expected_graph_values:
        raise ValueError('Generator output truncated while reading graph adjacency.')

    for i in range(expected_graph_values):
        graph_values.append(int(tokens[1 + i]))

    graph = [graph_values[i * players:(i + 1) * players] for i in range(players)]
    idx = 1 + expected_graph_values
    payoffs = [[None] * players for _ in range(players)]
    scale = 1

    def parse_value(tok):
        nonlocal scale
        if '.' in tok:
            dec = tok.split('.', 1)[1].rstrip('0')
            if dec:
                scale = max(scale, 10 ** len(dec))
        return float(tok)

    for i in range(players):
        for j in range(i + 1, players):
            if graph[i][j] == 0 and graph[j][i] == 0:
                continue

            if idx + 1 >= len(tokens):
                raise ValueError('Unexpected end of tokens while reading payoff sizes.')

            nrows = int(tokens[idx]); ncols = int(tokens[idx + 1])
            idx += 2
            expected = nrows * ncols
            if idx + expected > len(tokens):
                raise ValueError('Unexpected end of tokens while reading payoff matrix.')

            matrix_ij = [
                [parse_value(tokens[idx + r * ncols + c]) for c in range(ncols)]
                for r in range(nrows)
            ]
            idx += expected

            nrows2 = ncols
            ncols2 = nrows
            expected2 = nrows2 * ncols2
            if idx + expected2 > len(tokens):
                raise ValueError('Unexpected end of tokens while reading second payoff matrix.')

            matrix_ji = [
                [parse_value(tokens[idx + r * ncols2 + c]) for c in range(ncols2)]
                for r in range(nrows2)
            ]
            idx += expected2

            payoffs[i][j] = matrix_ij
            payoffs[j][i] = matrix_ji

    return players, graph, payoffs, scale


def write_polymatrix_format(path, players, payoffs, scale):
    actions = [None] * players
    for i in range(players):
        for j in range(players):
            if i == j:
                continue
            if payoffs[i][j] is not None:
                actions[i] = len(payoffs[i][j])
                break
            if payoffs[j][i] is not None:
                actions[i] = len(payoffs[j][i][0])
                break
        if actions[i] is None:
            raise ValueError(f'Could not infer number of actions for player {i + 1}.')

    max_abs = 0.0
    for i in range(players):
        for j in range(players):
            if i == j:
                continue
            matrix = payoffs[i][j]
            if matrix is None:
                continue
            for row in matrix:
                for value in row:
                    max_abs = max(max_abs, abs(value * scale))

    effective_scale = scale
    if max_abs > 0 and max_abs > TARGET_PAYOFF_MAX:
        effective_scale = scale * TARGET_PAYOFF_MAX / max_abs

    with open(path, 'w') as f:
        f.write('# m= \n')
        f.write(' '.join(str(a) for a in actions) + '\n')
        f.write('# A= \n\n')

        for i in range(players):
            for j in range(players):
                if i == j:
                    continue

                matrix = payoffs[i][j]
                if matrix is None:
                    matrix = [[0.0] * actions[j] for _ in range(actions[i])]

                f.write(f'# A{i+1}{j+1} \n')
                f.write('[ \n')
                for row in matrix:
                    row_line = ' '.join(
                        str(int(round(value * effective_scale)))
                        for value in row
                    )
                    f.write(row_line + '\n')
                f.write('] \n')


def generate_one_game(index, spec):
    name, graph, actions, g1, g2, p = spec
    output_name = PMGAMES_DIR / f'Savani_{index}.txt'
    print(f'Generating {output_name.name} from {name} ({graph})...')

    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        command = [
            str(PMGEN_BINARY),
            '-g', name,
            '-G', graph,
            '-a', str(actions),
            '-m', str(g1),
            '-n', str(g2),
            '-p', str(p),
            '-f', str(tmp_path),
            '-r', str(index * 12345),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        players, graph_mat, payoffs, scale = parse_pm_gen_file(tmp_path)
        write_polymatrix_format(output_name, players, payoffs, scale)

    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def generate_savani_games():
    PMGAMES_DIR.mkdir(exist_ok=True)
    ensure_pm_gen()
    for idx, spec in enumerate(GAME_SPECS, start=1):
        generate_one_game(idx, spec)


if __name__ == '__main__':
    generate_savani_games()
