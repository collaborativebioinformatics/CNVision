#!/usr/bin/env Rscript

# ---- Parse arguments ----
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: ... | Rscript svlen_hist_stdin.R <output_basename>\n")
  quit(status = 1)
}

out_base <- args[1]
out_pdf  <- paste0(out_base, ".pdf")
out_png  <- paste0(out_base, ".png")

# ---- Read lengths from stdin ----
lens <- scan(file("stdin"), what = numeric(), quiet = TRUE)

if (length(lens) == 0) {
  cat("No lengths provided on stdin.\n")
  quit(status = 1)
}

# ---- Clean and filter ----
lens <- abs(lens)        # absolute values
lens <- lens[lens > 100] # keep only >100 bp

if (length(lens) == 0) {
  cat("No lengths > 100 bp provided.\n")
  quit(status = 1)
}

# ---- Prepare log10 values ----
logx <- log10(lens)

# ---- Build histogram ----
h <- hist(logx, breaks = 100, plot = FALSE)

# ---- Plot function (so we can reuse for PDF and PNG) ----
plot_hist <- function(h) {
  plot(h$mids, h$counts,
       log = "xy", type = "h", lwd = 2, lend = 2,
       axes = FALSE,
       main = "Histogram of SV lengths (>100 bp)",
       xlab = "", ylab = "")
  
  # Custom x-axis ticks (lengths in bp)
  xmax <- ceiling(max(logx))
  x_ticks <- 2:xmax
  axis(1, at = x_ticks, labels = paste0(10^x_ticks / ifelse(10^x_ticks < 1000, 1, ifelse(10^x_ticks < 1e6, 1000, 1e6)),
                                        ifelse(10^x_ticks < 1000, "","k")))
  # Simpler option if you want only 100, 1k, 10k, 100k, 1M:
  # axis(1, at = c(2,3,4,5,6), labels = c("100", "1k", "10k", "100k", "1M"))
  
  # Custom y-axis ticks (counts in log scale)
  y_ticks <- 10^(0:ceiling(log10(max(h$counts))))
  axis(2, at = y_ticks, labels = format(y_ticks, scientific = FALSE), las = 1)
  
  # Axis labels
  mtext("SV length (bp)", side = 1, line = 2.5)
  mtext("Count", side = 2, line = 2.5)
}

# ---- Save PDF ----
pdf(out_pdf, width = 7, height = 5)
plot_hist(h)
dev.off()

# ---- Save high-resolution PNG ----
png(out_png, width = 7, height = 5, units = "in", res = 300)
plot_hist(h)
dev.off()

cat("Histogram saved to:\n", out_pdf, "\n", out_png, "\n")
