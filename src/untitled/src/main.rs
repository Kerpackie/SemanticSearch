use std::collections::HashSet;
use std::cmp::{max, min};
use std::time::Instant;
use std::fs;

#[derive(Debug, Clone, Copy)]
pub struct Part1Input {
    pub start: u64,
    pub end: u64,
}

/// ParseInput: turns the input text into a list of ranges.
pub fn parse_input(input: &str) -> Vec<Part1Input> {
    input
        .split(',')
        .filter_map(|range| {
            let bounds: Vec<&str> = range.split('-').collect();
            if bounds.len() == 2 {
                let start = bounds[0].parse::<u64>().ok()?;
                let end = bounds[1].parse::<u64>().ok()?;
                Some(Part1Input { start, end })
            } else {
                None
            }
        })
        .collect()
}

/// PartOne: brute-force approach.
pub fn part_one(input_list: &[Part1Input]) -> u64 {
    let mut tracker = 0;

    for item in input_list {
        for i in item.start..=item.end {
            let s = i.to_string();
            if s.len() % 2 == 0 && has_repeat(&s) {
                tracker += i;
            }
        }
    }

    tracker
}

/// HasRepeat: checks whether the decimal string is exactly the first half repeated as the second half.
fn has_repeat(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let half_len = s.len() / 2;
    let (first, second) = s.split_at(half_len);

    first == second
}

/// PartOneOptimised: Mathematical approach using aggregation.
pub fn part_one_optimised(input_list: &[Part1Input]) -> u64 {
    input_list
        .iter()
        .map(|item| sum_repeats_in_range(item.start, item.end))
        .sum()
}

/// SumRepeatsInRange: fast solver for summing numbers that are a "seed" repeated twice.
fn sum_repeats_in_range(range_start: u64, range_end: u64) -> u64 {
    let mut sum = 0;

    // Iterate half-lengths from 1 to 9 (since u64 max is ~18-19 digits)
    for half_len in 1..=9 {
        let power_of_10 = 10_u64.pow(half_len);
        let multiplier = power_of_10 + 1;

        // Valid seed range logic
        let min_seed_limit = power_of_10 / 10;
        let max_seed_limit = power_of_10 - 1;

        // Lowest seed that produces a number >= rangeStart
        // Math.Ceiling(rangeStart / multiplier) equivalent:
        // Using explicit remainder check to be consistent with Part 2 safety
        let mut start_seed = range_start / multiplier;
        if range_start % multiplier != 0 {
            start_seed += 1;
        }

        let end_seed = range_end / multiplier;

        // Intersect limits with valid seed range
        let effective_start = max(min_seed_limit, start_seed);
        let effective_end = min(max_seed_limit, end_seed);

        // If we have seeds in range, sum them
        if effective_start <= effective_end {
            let count = effective_end - effective_start + 1;

            // Arithmetic progression sum: (first + last) * count / 2
            let sum_of_seeds = (effective_start + effective_end) * count / 2;

            sum += sum_of_seeds * multiplier;
        }

        // Optimization: if the smallest number for this length is beyond our range, stop.
        if let Some(val) = min_seed_limit.checked_mul(multiplier) {
            if val > range_end {
                break;
            }
        }
    }

    sum
}

pub fn solve_part_two(input_list: &[Part1Input]) -> u64 {
    // HashSet prevents double counting numbers
    let mut unique_invalid_ids = HashSet::new();

    for range in input_list {
        collect_patterns(range.start, range.end, &mut unique_invalid_ids);
    }

    unique_invalid_ids.iter().sum()
}

fn collect_patterns(range_start: u64, range_end: u64, results: &mut HashSet<u64>) {
    // 1. Iterate over Seed Length (1 digit to 9 digits)
    // Max is 9, because a 10-digit seed x 2 reps = 20 digits (Overflows u64)
    for seed_len in 1..=9 {
        let seed_min = 10_u64.pow(seed_len - 1);
        let seed_max = 10_u64.pow(seed_len) - 1;

        let mut current_multiplier = 1_u64;
        let shift = 10_u64.pow(seed_len);

        // 2. Iterate Repetitions (Start at 2)
        // We use pure integer checks to prevent overflow loops
        loop {
            // Check for overflow before multiplying:
            // if (currentMultiplier * shift + 1) would overflow
            // Rust equivalent of: if (u64::MAX / shift < current_multiplier) break;
            match current_multiplier.checked_mul(shift) {
                Some(val) => {
                    // Also check the +1
                    if let Some(new_mult) = val.checked_add(1) {
                        current_multiplier = new_mult;
                    } else {
                        break;
                    }
                },
                None => break,
            }

            // 3. Integer-only Intersection Logic

            // Calculate Min Seed: Ceiling(rangeStart / currentMultiplier)
            // Safer manual ceiling to avoid (A+B) overflow risks
            let mut min_calculated_seed = range_start / current_multiplier;
            if range_start % current_multiplier != 0 {
                min_calculated_seed += 1;
            }

            // Calculate Max Seed: Floor(rangeEnd / currentMultiplier)
            let max_calculated_seed = range_end / current_multiplier;

            // Clamp to valid seed range (e.g. 10..99)
            let start_seed = max(seed_min, min_calculated_seed);
            let end_seed = min(seed_max, max_calculated_seed);

            // 4. Collect Valid Numbers
            if start_seed <= end_seed {
                for s in start_seed..=end_seed {
                    // Safe to multiply because s <= max_calculated_seed which was derived from range_end
                    results.insert(s * current_multiplier);
                }
            }

            // Optimization: If the smallest possible number for this repetition count
            // is already beyond our range, we can stop adding repetitions.
            // Check for potential overflow in the check itself:
            match seed_min.checked_mul(current_multiplier) {
                Some(val) => {
                    if val > range_end {
                        break;
                    }
                },
                None => break, // Overflow means it's definitely bigger than range_end
            }
        }
    }
}

pub fn main() {
    println!("--- Advent of Code - Day 2 Benchmarks ---");

    // 0. Load Phase
    println!("Loading data from 'input.txt'...");

    // Read directly from file.
    // Panics if file is missing, which is standard for AoC solutions.
    let raw_input = fs::read_to_string("input.txt")
        .expect("Failed to read 'input.txt'. Please ensure the file exists in the current directory.");

    // Parse data before starting the benchmark timer to ensure fairness
    let input_data = parse_input(&raw_input);

    println!("Loaded {} input ranges for testing.", input_data.len());
    println!("-----------------------------------------------------");

    // 1. Benchmark Part One (Brute Force)
    let t0 = Instant::now();
    let res_p1 = part_one(&input_data);
    let d_p1 = t0.elapsed();
    println!("Part 1 (Brute Force)  -> Result: {:<15} | Time: {:.2?}", res_p1, d_p1);

    // 2. Benchmark Part One (Optimized)
    let t1 = Instant::now();
    let res_p1_opt = part_one_optimised(&input_data);
    let d_p1_opt = t1.elapsed();
    println!("Part 1 (Optimized)    -> Result: {:<15} | Time: {:.2?}", res_p1_opt, d_p1_opt);

    // Verification
    if res_p1 != res_p1_opt {
        eprintln!("ERROR: Part 1 mismatch! Brute: {}, Opt: {}", res_p1, res_p1_opt);
    }

    // 3. Benchmark Part Two (Solve Part Two)
    let t2 = Instant::now();
    let res_p2 = solve_part_two(&input_data);
    let d_p2 = t2.elapsed();
    println!("Part 2 (SolvePartTwo) -> Result: {:<15} | Time: {:.2?}", res_p2, d_p2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_repeat() {
        assert!(has_repeat("1212"));
        assert!(!has_repeat("123"));
        assert!(!has_repeat("1213"));
    }

    #[test]
    fn test_math_logic_small() {
        let input = vec![Part1Input { start: 10, end: 1500 }];
        let brute = part_one(&input);
        let opt = part_one_optimised(&input);
        assert_eq!(brute, opt);
    }
}