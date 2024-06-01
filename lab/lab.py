import subprocess
from collections import Counter
import math
from scipy.stats import chi2
import matplotlib.pyplot as plt


def plot_graph(x, y, title='My Graph', x_label='X-axis', y_label='Y-axis'):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


# Перевод в диапозон от 0 до 1
def st(a: float, b: float, nums: list[float]) -> list[float]:
    res_nums = []
    m = max(nums) + 1
    for x in nums:
        res_nums.append(x / m * b + a)
    return res_nums


# Мат ожидание
def mean(nums: list[float]) -> float:
    return sum(nums) / len(nums)


# Среднеквадратичное отклонение
def st_dev(nums: list[float]) -> float:
    m = mean(nums)
    return math.sqrt(sum((x - m) ** 2 for x in nums) / (len(nums) - 1))


def relative_errors(nums: list[float]) -> None:
    expected_mean = 0.5
    expected_std_dev = 0.28
    current_mean = mean(nums)
    current_std_dev = st_dev(nums)
    print(f"Погрешность мат ожидания: "
          f"{math.fabs(expected_mean - current_mean)}")
    print(f"Погрешность среднеквадратичного отклонения: "
          f"{math.fabs(expected_std_dev - current_std_dev)}")


def build_graph(nums: list[float]) -> None:
    n = len(nums)
    means = []
    st_devs = []
    volumes = []
    while n > 1:
        means.append(mean(nums[:n]))
        st_devs.append(st_dev(nums[:n]))
        volumes.append(n)
        n //= 2
    plot_graph(
        volumes,
        means,
        "График зависимости математического ожидания "
        "от объёма выборки",
        "Объём выборки",
        "Математическое ожидание"
    )
    plot_graph(
        volumes,
        st_devs,
        "График зависимости "
        "среднеквадратичного отклонения "
        "от объёма выборки",
        "Объём выборки",
        "Среднеквадратичное отклонение"
    )


# Хи квадрат
def chi_square(
    nums: list[float],
    alpha: float = 0.05,
    observed: list[float] = None,
    expected: list[float] = None

) -> bool:
    if observed is None:
        observed_dict = Counter(nums)
        sorted_num_set = sorted(observed_dict)
        observed = [observed_dict[num] for num in sorted_num_set]
        observed[0] = observed[0] + observed[1]
        observed.pop(1)
        expected = []
        for i in range(1, len(sorted_num_set)):
            expected.append(
                (sorted_num_set[i] - sorted_num_set[i - 1]) * len(nums)
            )
    chi2_res = sum((o - e) * (o - e) / e for o, e in zip(observed, expected))
    chi2_quantile = chi2.ppf(1 - alpha,  len(observed) - 1)
    return chi2_res <= chi2_quantile


def series(nums: list[float], alpha: float = 0.05, d: int = 32) -> bool:
    observed = [0] * (d ** 2)
    for j in range(len(nums) // 2):
        q = math.floor(nums[2 * j] * d)
        r = math.floor(nums[2 * j + 1] * d)
        observed[q * d + r] += 1
    expected = [len(nums) / (d * d)] * d * d
    return chi_square(
        nums=nums, expected=expected, observed=observed, alpha=alpha
    )


def interval(
        nums: list[float],
        alpha: float = 0.05,
        t: int = 10,
        a: float = 0.5,
        b: float = 1
) -> bool:
    n = len(nums)
    j = -1
    s = 0
    c = [0] * (t + 1)
    while s < n:
        r = 0
        j += 1
        while j < len(nums) and a <= nums[j] <= b:
            r += 1
            j += 1
        c[min(r, t)] += 1
        s += 1
    p = b - a
    expected = [
       n * p * pow(1.0 - p, r) for r in range(t)
    ] + [n * pow(1.0 - p, t)]
    return chi_square(nums=nums, alpha=alpha, observed=c, expected=expected)


def stirling(n: int, k: int) -> float:
    s = [[0] * (k + 1) for _ in range(n + 1)]
    s[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            s[i][j] = j * s[i - 1][j] + s[i - 1][j - 1]

    return float(s[n][k])


# Критерий разбиений
def partitions(
        nums: list[float],
        d: int = 16,
        k: int = 2,
        alpha: float = 0.05
) -> bool:
    n = len(nums)
    observed = [0] * k
    for i in range(0, n // k, k):
        unique_elems = list(set([math.floor(j * d) for j in nums[i: i + k]]))
        observed[len(unique_elems) - 1] += 1
    expected = [0] * k
    for r in range(1, k + 1):
        p = 1.0
        for i in range(r):
            p *= d - i
        expected[r - 1] = (n / k) * (p / pow(d, k)) * stirling(k, r)
    return chi_square(
        nums=nums, observed=observed, expected=expected, alpha=alpha
    )


# Критерий перестановок
def permutations(nums, alpha=0.05, t=3) -> bool:
    k = math.factorial(t)
    groups = {}
    for i in range(0, len(nums), t):
        group = tuple(sorted(nums[i:i + t]))
        groups[group] = groups.get(group, 0) + 1
    observed = list(groups.values())
    expected = [len(nums) / k] * len(observed)
    return chi_square(
        nums=nums, observed=observed, expected=expected, alpha=alpha
    )


def monotony(
        nums: list[float],
        alpha: float = 0.05
) -> bool:
    a = [
        [4529.4, 9044.9, 13568.0, 18091.0, 22615.0, 27892.0],
        [9044.9, 18097.0, 27139.0, 36187.0, 45234.0, 55789.0],
        [13568.0, 27139.0, 40721.0, 54281.0, 67852.0, 83685.0],
        [18091.0, 36187.0, 54281.0, 72414.0, 90470.0, 111580.0],
        [22615.0, 45234.0, 67852.0, 90470.0, 113262.0, 139476.0],
        [27892.0, 55789.0, 83685.0, 111580.0, 139476.0, 172860.0]
    ]
    b = [
        1.0 / 6.0,
        5.0 / 24.0,
        11.0 / 120.0,
        19.0 / 720.0,
        29.0 / 5040.0,
        1.0 / 840.0
    ]
    c_array = []
    i = 0
    while i < len(nums):
        j = 1
        while i + j < len(nums) and nums[i + j - 1] <= nums[i + j]:
            j += 1
        c_array.append(j)
        i += j
    m, group_counts = 0, {}
    for length in c_array:
        m = max(m, length)
        group_counts[length] = group_counts.get(length, 0) + 1
    statistics = []
    offset = 0
    n = len(nums)
    for c_len in c_array:
        m = 0
        min_val = min(c_len, 6)
        for i in range(min_val):
            for j in range(min_val):
                m += (
                             nums[i + offset] - n * b[i]
                     ) * (nums[j + offset] - n * b[j]) * a[i][j]
        offset += c_len
        statistics.append(m)
    num_intervals = 6
    observed = [0] * num_intervals
    for num in st(0, 1, statistics):
        if 0 <= num <= 1:
            interval_index = int(num * num_intervals)
            if interval_index == num_intervals:
                interval_index -= 1
            observed[interval_index] += 1
    expected = [len(nums) / 6] * 6
    return chi_square(
        nums=nums, observed=observed, expected=expected, alpha=alpha
    )


def conflicts(nums, alpha: float = 0.05, m: int = 1024):
    n = len(nums)
    conflicts_p = (n - len(set(nums))) / n
    p0 = 1 - n / m + math.comb(n, 2) / (m * m)
    avg_conflicts_p = n / m - 1 + p0
    return conflicts_p <= avg_conflicts_p / alpha


commands = {
    "ls": "prng.exe /g:lc 257 8 9 7",
    "add": "prng.exe /g:add 257 3 7 1 2 3 4 5 6 7 8",
    "5p": "prng.exe /g:5p 7 2 4 6 10 0b1001011",
    "lfsr": "prng.exe /g:lfsr 0b1011011 0b1001011",
    "nfsr": "prng.exe /g:nfsr 0b1001011 0b1011010 0b1001011 10 83 45 67",
    "mt": "prng.exe /g:mt 1000 42",
    "rc4": "prng.exe /g:rc4 0 527 30 557 60 587 90 617 120 647 150 677 180 "
          "707 210 737 240 767 270 797 300 827 330 857 360 887 390 917 420 "
          "947 450 977 480 1007 510 13 540 43 570 73 600 103 630 133 660 "
          "163 690 193 720 223 750 253 780 283 810 313 840 343 870 373 "
          "900 403 930 433 960 463 990 493 1020 523 26 553 56 583 86 613 "
          "116 643 146 673 176 703 206 733 236 763 266 793 296 823 326 853 "
          "356 883 386 913 416 943 446 973 476 1003 506 9 53 6 39 566 69 "
          "596 99 626 129 656 159 686 189 716 219 746 249 776 279 806 309 "
          "836 339 866 369 896 399 926 429 956 459 986 4 89 1016 519 22 "
          "549 52 579 82 609 112 639 142 669 172 699 202 729 232 759 262 "
          "789 292 819 322 849 352 879 382 909 412 939 442 969 472 999 502 "
          "5 532 35 562 65 592 95 622 125 652 155 682 185 712 215 742 245 "
          "772 275 802 305 832 335 862 365 892 395 922 425 952 455 982 485 "
          "1012 515 18 545 48 575 78 605 108 635 138 665 168 695 198 725 "
          "228 755 258 785 288 815 318 845 348 875 378 905 408 935 438 965 "
          "468 995 498 1 528 31 558 61 588 91 618 121 648 151 678 181 708 "
          "211 738 241",
    "rsa": "prng.exe /g:rsa 30824219905435791457998495079 17 10 11",
    "bbs": "prng.exe /g:bbs 17"
}

for i in commands:
    print(f"----------------------{i}---------------------------")
    subprocess.run(commands[i], shell=True, capture_output=True, text=True)
    with open('rnd.dat', 'r') as file:
        my_nums = st(
            0, 1, list(map(int, file.readline()[:-1].split(',')))[:10000]
        )
    print("Мат ожидание:")
    print(mean(my_nums))
    print("Среднеквадратичное отклонение:")
    print(st_dev(my_nums))
    relative_errors(my_nums)
    build_graph(my_nums)
    print("Критерий хи-квадрат:")
    print(chi_square(my_nums))
    print("Критерий серий:")
    print(series(my_nums))
    print("Критерий интервалов:")
    print(interval(my_nums))
    print("Критерий разбиений:")
    print(partitions(my_nums))
    print("Критерий перестановок:")
    print(permutations(my_nums))
    print("Критерий монотонности:")
    print(monotony(my_nums))
    print("Критерий конфликтов:")
    print(conflicts(my_nums))
