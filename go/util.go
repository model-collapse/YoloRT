package main

//#include "api.h"
import "C"

type LabeledPerson struct {
	Loc        []int32  `json:"loc"`
	Activities []string `json:"activities"`
}

func parseDetectionResultFromC(ctx *context_t) (ret []LabeledPerson) {
	for i := 0; i < ctx.num_boxes; i++ {
		p := LabeledPerson{Loc: make([]int32, 4)}
		p.Loc[0] = ctx.left
		p.Loc[1] = ctx.top
		p.Loc[2] = ctx.width
		p.Loc[3] = ctx.height

		for i := 0; i < C.NUM_ACTIVITIES && ctx.activities[i][0] != C.char(0); i++ {
			actName := C.GoString(ctx.activities[i])
			p.Activities = append(p.Activities, actName)
		}

		ret = append(ret, p)
	}

	return
}
