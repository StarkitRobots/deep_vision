library(ggplot2)

args <- commandArgs(TRUE)


if (length(args) < 1) {
    cat("Usage: ... <resultFile (csv)>\n");
    quit(status=1)
}

data <- read.csv(args[1])

g <- ggplot(data, aes(x=acceptance_score))
g <- g + geom_line(aes(y=false_pos_rate), color='red')
g <- g + geom_line(aes(y=false_neg_rate), color='black')

ggsave("graph.png")
