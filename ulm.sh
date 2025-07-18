for limit in c d f l m n q s t u v x; do
	ulimit -S -$limit $(ulimit -H -$limit)
done
